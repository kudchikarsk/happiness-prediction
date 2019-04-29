from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField, TextAreaField
from wtforms.validators import DataRequired, Length, Email, EqualTo

class ReusableForm(FlaskForm):
    text = TextAreaField('Text to analyse',
                           validators=[DataRequired()])
    
    submit = SubmitField('Predict')